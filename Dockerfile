# Use an official Node runtime as the base image
FROM node:18-alpine AS builder

# Set the working directory in the container
WORKDIR /app

# Install dependencies required for node-gyp and other build tools
RUN apk add --no-cache \
    python3 \
    make \
    g++ \
    git

# Copy package files
COPY package*.json ./

# Install npm@10.9.2 globally to address version mismatch
RUN npm install -g npm@10.9.2

# Install dependencies with legacy peer deps flag to avoid conflicts
RUN npm ci --legacy-peer-deps

# Copy the rest of the application code
COPY . .

# Build the Next.js app
RUN npm run build

# Start a new stage for a smaller production image
FROM node:18-alpine

WORKDIR /app

# Copy built assets from the builder stage
COPY --from=builder /app/.next ./.next
COPY --from=builder /app/node_modules ./node_modules
COPY --from=builder /app/package.json ./package.json
COPY --from=builder /app/public ./public

# Expose the port the app runs on
EXPOSE 3000

# Define the command to run the app
CMD ["npm", "start"]




#docker run -p 3000:3000 maritime-ai-control-system